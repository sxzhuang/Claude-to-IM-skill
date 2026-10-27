/**
 * Codex Provider — LLMProvider implementation backed by @openai/codex-sdk.
 *
 * Maps Codex SDK thread events to the SSE stream format consumed by
 * the bridge conversation engine, making Codex a drop-in alternative
 * to the Claude Code SDK backend.
 *
 * Requires `@openai/codex-sdk` to be installed (optionalDependency).
 * The provider lazily imports the SDK at first use and throws a clear
 * error if it is not available.
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import readline from 'node:readline';
import { createRequire as createCodexRequire } from 'node:module';

import type { LLMProvider, StreamChatParams } from 'claude-to-im/src/lib/bridge/host.js';
import type { PendingPermissions } from './permission-gateway.js';
import { sseEvent } from './sse-utils.js';

/** MIME → file extension for temp image files. */
const MIME_EXT: Record<string, string> = {
  'image/png': '.png',
  'image/jpeg': '.jpg',
  'image/jpg': '.jpg',
  'image/gif': '.gif',
  'image/webp': '.webp',
};

// All SDK types kept as `any` because @openai/codex-sdk is optional.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CodexModule = any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CodexInstance = any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ThreadInstance = any;

/**
 * Map bridge permission modes to Codex approval policies.
 * - 'acceptEdits' (code mode) → 'on-failure' (auto-approve most things)
 * - 'plan' → 'on-request' (ask before executing)
 * - 'default' (ask mode) → 'on-request'
 */
function toApprovalPolicy(permissionMode?: string): string {
  switch (permissionMode) {
    case 'acceptEdits': return 'on-failure';
    case 'plan': return 'on-request';
    case 'default': return 'on-request';
    default: return 'on-request';
  }
}

/** Whether to forward bridge model to Codex CLI. Default: false (use Codex current/default model). */
function shouldPassModelToCodex(): boolean {
  return process.env.CTI_CODEX_PASS_MODEL === 'true';
}

/** Allow Codex to run outside a trusted Git repository when explicitly enabled. */
function shouldSkipGitRepoCheck(): boolean {
  return process.env.CTI_CODEX_SKIP_GIT_REPO_CHECK === 'true';
}

/** Build the environment inherited by Codex without proxying the IM bridge itself. */
function codexProcessEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  const proxyUrl = process.env.CTI_CODEX_PROXY_URL;
  if (!proxyUrl) return env;

  env.HTTP_PROXY = proxyUrl;
  env.HTTPS_PROXY = proxyUrl;
  env.ALL_PROXY = proxyUrl;
  env.http_proxy = proxyUrl;
  env.https_proxy = proxyUrl;
  env.all_proxy = proxyUrl;

  const noProxy = process.env.CTI_CODEX_NO_PROXY || 'localhost,127.0.0.1';
  env.NO_PROXY = noProxy;
  env.no_proxy = noProxy;
  return env;
}

export class CodexProvider implements LLMProvider {
  private sdk: CodexModule | null = null;
  private codex: CodexInstance | null = null;

  /** Maps session IDs to Codex thread IDs for resume. */
  private threadIds = new Map<string, string>();

  constructor(private pendingPerms: PendingPermissions) {}

  /** Read Codex's latest context-window usage for the status command. */
  async getContextStatus(params: { sessionId: string; sdkSessionId?: string }): Promise<{
    contextLeftPercent: number;
    tokensUsed: number;
    modelContextWindow: number;
  } | null> {
    const threadId = params.sdkSessionId;
    if (!threadId) return null;
    let executable = process.env.CTI_CODEX_EXECUTABLE;
    let executableArgs: string[] = ['app-server', '--stdio'];
    if (!executable) {
      try {
        executable = process.execPath;
        executableArgs = [createCodexRequire(import.meta.url)('@openai/codex/bin/codex.js'), ...executableArgs];
      } catch {
        executable = 'codex';
      }
    }
    const child = spawn(executable, executableArgs, { stdio: ['pipe', 'pipe', 'pipe'], env: codexProcessEnv() });

    return await new Promise((resolve, reject) => {
      let settled = false;
      let initialized = false;
      let responseReceived = false;
      let usage: { tokensUsed: number; modelContextWindow: number } | null = null;
      let stderr = '';
      let timeout: NodeJS.Timeout;
      const finish = (error?: Error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        try { child.kill('SIGTERM'); } catch { /* best effort */ }
        if (error) reject(error);
        else if (!usage) resolve(null);
        else resolve({ ...usage, contextLeftPercent: Math.max(0, Math.min(100, (1 - usage.tokensUsed / usage.modelContextWindow) * 100)) });
      };
      const maybeFinish = () => { if (responseReceived && usage) finish(); };
      timeout = setTimeout(() => finish(new Error('Codex context status timed out after 30 seconds')), 30_000);
      const send = (message: Record<string, unknown>) => child.stdin.write(`${JSON.stringify(message)}\n`);
      const lines = readline.createInterface({ input: child.stdout });
      lines.on('line', (line) => {
        if (settled || !line.trim()) return;
        let message: Record<string, unknown>;
        try { message = JSON.parse(line) as Record<string, unknown>; } catch { return; }
        if (message.id === 1 && !initialized) {
          initialized = true;
          send({ method: 'initialized', params: {} });
          send({ id: 2, method: 'thread/resume', params: { threadId } });
        } else if (message.id === 2) {
          if (message.error) {
            const error = message.error as { message?: string };
            finish(new Error(error.message || 'Codex context status request failed'));
          } else {
            responseReceived = true;
            maybeFinish();
          }
        } else if (message.method === 'thread/tokenUsage/updated') {
          const tokenUsage = (message.params as { tokenUsage?: { last?: { totalTokens?: number }; modelContextWindow?: number | null } } | undefined)?.tokenUsage;
          const tokensUsed = tokenUsage?.last?.totalTokens;
          const modelContextWindow = tokenUsage?.modelContextWindow;
          if (typeof tokensUsed === 'number' && typeof modelContextWindow === 'number' && modelContextWindow > 0) {
            usage = { tokensUsed, modelContextWindow };
            maybeFinish();
          }
        }
      });
      child.stderr.on('data', (chunk: Buffer | string) => { stderr = (stderr + chunk.toString()).slice(-4096); });
      child.once('error', (error) => finish(error));
      child.once('close', (code) => {
        if (!settled) {
          const detail = stderr.trim() ? `: ${stderr.trim()}` : '';
          finish(new Error(`Codex app-server exited with code ${code ?? 'unknown'}${detail}`));
        }
      });
      send({ id: 1, method: 'initialize', params: {
        clientInfo: { name: 'claude-to-im', title: 'Claude-to-IM', version: '0.1.0' },
        capabilities: { experimentalApi: true },
      } });
    });
  }

  /**
   * Lazily load the Codex SDK. Throws a clear error if not installed.
   */
  private async ensureSDK(): Promise<{ sdk: CodexModule; codex: CodexInstance }> {
    if (this.sdk && this.codex) {
      return { sdk: this.sdk, codex: this.codex };
    }

    try {
      this.sdk = await (Function('return import("@openai/codex-sdk")')() as Promise<CodexModule>);
    } catch {
      throw new Error(
        '[CodexProvider] @openai/codex-sdk is not installed. ' +
        'Install it with: npm install @openai/codex-sdk'
      );
    }

    // Resolve API key: CTI_CODEX_API_KEY > CODEX_API_KEY > OPENAI_API_KEY > (login auth)
    const apiKey = process.env.CTI_CODEX_API_KEY
      || process.env.CODEX_API_KEY
      || process.env.OPENAI_API_KEY
      || undefined;
    const baseUrl = process.env.CTI_CODEX_BASE_URL || undefined;

    const CodexClass = this.sdk.Codex;
    this.codex = new CodexClass({
      env: codexProcessEnv(),
      ...(apiKey ? { apiKey } : {}),
      ...(baseUrl ? { baseUrl } : {}),
    });

    return { sdk: this.sdk, codex: this.codex };
  }

  streamChat(params: StreamChatParams): ReadableStream<string> {
    const self = this;

    return new ReadableStream<string>({
      start(controller) {
        (async () => {
          const tempFiles: string[] = [];
          try {
            const { codex } = await self.ensureSDK();

            // Resolve or create thread
            const inMemoryThreadId = self.threadIds.get(params.sessionId);
            const savedThreadId = inMemoryThreadId || params.sdkSessionId || undefined;

            const approvalPolicy = toApprovalPolicy(params.permissionMode);
            const passModel = shouldPassModelToCodex();

            const threadOptions: Record<string, unknown> = {
              ...((passModel || params.modelOverride) && params.model ? { model: params.model } : {}),
              ...(params.reasoningEffort ? { modelReasoningEffort: params.reasoningEffort } : {}),
              ...(params.workingDirectory ? { workingDirectory: params.workingDirectory } : {}),
              ...(shouldSkipGitRepoCheck() ? { skipGitRepoCheck: true } : {}),
              approvalPolicy,
            };

            // Build input: Codex SDK UserInput supports { type: "text" } and
            // { type: "local_image", path: string }. We write base64 data to
            // temp files so the SDK can read them as local images.
            const imageFiles = params.files?.filter(
              f => f.type.startsWith('image/')
            ) ?? [];

            let input: string | Array<Record<string, string>>;
            if (imageFiles.length > 0) {
              const parts: Array<Record<string, string>> = [
                { type: 'text', text: params.prompt },
              ];
              for (const file of imageFiles) {
                const ext = MIME_EXT[file.type] || '.png';
                const tmpPath = path.join(os.tmpdir(), `cti-img-${Date.now()}-${Math.random().toString(36).slice(2)}${ext}`);
                fs.writeFileSync(tmpPath, Buffer.from(file.data, 'base64'));
                tempFiles.push(tmpPath);
                parts.push({ type: 'local_image', path: tmpPath });
              }
              input = parts;
            } else {
              input = params.prompt;
            }

            let thread: ThreadInstance;
            if (savedThreadId) {
              try {
                thread = codex.resumeThread(savedThreadId, threadOptions);
              } catch (err) {
                const message = err instanceof Error ? err.message : String(err);
                throw new Error(`Failed to resume Codex session ${savedThreadId}: ${message}`);
              }
            } else {
              thread = codex.startThread(threadOptions);
            }

            try {
              const { events } = await thread.runStreamed(input);

              for await (const event of events) {
                if (params.abortController?.signal.aborted) {
                  break;
                }

                switch (event.type) {
                    case 'thread.started': {
                      const threadId = event.thread_id as string;
                      if (savedThreadId && threadId !== savedThreadId) {
                        throw new Error(
                          `Codex resumed an unexpected session: requested ${savedThreadId}, received ${threadId}`
                        );
                      }
                      self.threadIds.set(params.sessionId, threadId);

                      controller.enqueue(sseEvent('status', {
                        session_id: threadId,
                      }));
                      break;
                    }

                    case 'item.completed': {
                      const item = event.item as Record<string, unknown>;
                      self.handleCompletedItem(controller, item);
                      break;
                    }

                    case 'turn.completed': {
                      const usage = event.usage as Record<string, unknown> | undefined;
                      const threadId = self.threadIds.get(params.sessionId);

                      controller.enqueue(sseEvent('result', {
                        usage: usage ? {
                          input_tokens: usage.input_tokens ?? 0,
                          output_tokens: usage.output_tokens ?? 0,
                          cache_read_input_tokens: usage.cached_input_tokens ?? 0,
                        } : undefined,
                        ...(threadId ? { session_id: threadId } : {}),
                      }));
                      break;
                    }

                    case 'turn.failed': {
                      const error = (event as { message?: string }).message;
                      controller.enqueue(sseEvent('error', error || 'Turn failed'));
                      break;
                    }

                    case 'error': {
                      const error = (event as { message?: string }).message;
                      controller.enqueue(sseEvent('error', error || 'Thread error'));
                      break;
                    }

                    // item.started, item.updated, turn.started — no action needed
                }
              }
            } catch (err) {
              if (savedThreadId) {
                const message = err instanceof Error ? err.message : String(err);
                if (message.startsWith('Codex resumed an unexpected session:')) {
                  throw err;
                }
                throw new Error(`Failed to resume Codex session ${savedThreadId}: ${message}`);
              }
              throw err;
            }

            controller.close();
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            console.error('[codex-provider] Error:', err instanceof Error ? err.stack || err.message : err);
            try {
              controller.enqueue(sseEvent('error', message));
              controller.close();
            } catch {
              // Controller already closed
            }
          } finally {
            // Clean up temp image files
            for (const tmp of tempFiles) {
              try { fs.unlinkSync(tmp); } catch { /* ignore */ }
            }
          }
        })();
      },
    });
  }

  /**
   * Map a completed Codex item to SSE events.
   */
  private handleCompletedItem(
    controller: ReadableStreamDefaultController<string>,
    item: Record<string, unknown>,
  ): void {
    const itemType = item.type as string;

    switch (itemType) {
      case 'agent_message': {
        const text = (item.text as string) || '';
        if (text) {
          controller.enqueue(sseEvent('text', text));
        }
        break;
      }

      case 'command_execution': {
        const toolId = (item.id as string) || `tool-${Date.now()}`;
        const command = item.command as string || '';
        const output = item.aggregated_output as string || '';
        const exitCode = item.exit_code as number | undefined;
        const isError = exitCode != null && exitCode !== 0;

        controller.enqueue(sseEvent('tool_use', {
          id: toolId,
          name: 'Bash',
          input: { command },
        }));

        const resultContent = output || (isError ? `Exit code: ${exitCode}` : 'Done');
        controller.enqueue(sseEvent('tool_result', {
          tool_use_id: toolId,
          content: resultContent,
          is_error: isError,
        }));
        break;
      }

      case 'file_change': {
        const toolId = (item.id as string) || `tool-${Date.now()}`;
        const changes = item.changes as Array<{ path: string; kind: string }> || [];
        const summary = changes.map(c => `${c.kind}: ${c.path}`).join('\n');

        controller.enqueue(sseEvent('tool_use', {
          id: toolId,
          name: 'Edit',
          input: { files: changes },
        }));

        controller.enqueue(sseEvent('tool_result', {
          tool_use_id: toolId,
          content: summary || 'File changes applied',
          is_error: false,
        }));
        break;
      }

      case 'mcp_tool_call': {
        const toolId = (item.id as string) || `tool-${Date.now()}`;
        const server = item.server as string || '';
        const tool = item.tool as string || '';
        const args = item.arguments as unknown;
        const result = item.result as { content?: unknown; structured_content?: unknown } | undefined;
        const error = item.error as { message?: string } | undefined;

        const resultContent = result?.content ?? result?.structured_content;
        const resultText = typeof resultContent === 'string' ? resultContent : (resultContent ? JSON.stringify(resultContent) : undefined);

        controller.enqueue(sseEvent('tool_use', {
          id: toolId,
          name: `mcp__${server}__${tool}`,
          input: args,
        }));

        controller.enqueue(sseEvent('tool_result', {
          tool_use_id: toolId,
          content: error?.message || resultText || 'Done',
          is_error: !!error,
        }));
        break;
      }

      case 'reasoning': {
        // Reasoning is internal; emit as status
        const text = (item.text as string) || '';
        if (text) {
          controller.enqueue(sseEvent('status', { reasoning: text }));
        }
        break;
      }
    }
  }
}
