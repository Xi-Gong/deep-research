import { createOpenAI, type OpenAIProviderSettings } from '@ai-sdk/openai';
import { getEncoding } from 'js-tiktoken';

import { RecursiveCharacterTextSplitter } from './text-splitter';

interface CustomOpenAIProviderSettings extends OpenAIProviderSettings {
  baseURL?: string;
}

// Providers，添加代理支持
const openai = createOpenAI({
  apiKey: process.env.OPENAI_KEY!,
  baseURL: process.env.OPENAI_ENDPOINT || 'https://api.openai.com/v1',
  fetch: process.env.PROXY_URL ? async (url, init) => {
    const { default: nodeFetch } = await import('node-fetch');
    const { HttpsProxyAgent } = await import('https-proxy-agent');
    const proxyAgent = new HttpsProxyAgent(process.env.PROXY_URL!);
    return nodeFetch(url, { ...init, agent: proxyAgent });
  } : undefined,
} as CustomOpenAIProviderSettings);

const customModelName = process.env.OPENAI_MODEL || 'o3-mini';

// 摘要模型设置
const summaryModelName = process.env.SUMMARY_MODEL || 'gpt-4o-mini';

// Models
export const o3MiniModel = openai(customModelName, {
  reasoningEffort: customModelName.startsWith('o3') ? 'medium' : undefined,
  structuredOutputs: true,
});

export const summaryModel = openai(summaryModelName, {
  reasoningEffort: summaryModelName.startsWith('o3') ? 'medium' : undefined,
  structuredOutputs: true,
});

const MinChunkSize = 140;
const encoder = getEncoding('o200k_base');

// trim prompt to maximum context size
export function trimPrompt(
  prompt: string,
  contextSize = Number(process.env.CONTEXT_SIZE) || 128_000,
) {
  if (!prompt) {
    return '';
  }

  const length = encoder.encode(prompt).length;
  if (length <= contextSize) {
    return prompt;
  }

  const overflowTokens = length - contextSize;
  // on average it's 3 characters per token, so multiply by 3 to get a rough estimate of the number of characters
  const chunkSize = prompt.length - overflowTokens * 3;
  if (chunkSize < MinChunkSize) {
    return prompt.slice(0, MinChunkSize);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap: 0,
  });
  const trimmedPrompt = splitter.splitText(prompt)[0] ?? '';

  // last catch, there's a chance that the trimmed prompt is same length as the original prompt, due to how tokens are split & innerworkings of the splitter, handle this case by just doing a hard cut
  if (trimmedPrompt.length === prompt.length) {
    return trimPrompt(prompt.slice(0, chunkSize), contextSize);
  }

  // recursively trim until the prompt is within the context size
  return trimPrompt(trimmedPrompt, contextSize);
}
