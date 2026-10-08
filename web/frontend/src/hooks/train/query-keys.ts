export const trainKeys = {
  all: () => ['train'] as const,
  lists: () => [...trainKeys.all(), 'list'] as const,
  logs: (jobId: string) => [...trainKeys.all(), 'logs', jobId] as const,
  result: (jobId: string) => [...trainKeys.all(), 'result', jobId] as const,
};
