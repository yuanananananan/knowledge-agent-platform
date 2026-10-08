export const valKeys = {
  all: () => ['val'] as const,
  lists: () => [...valKeys.all(), 'list'] as const,
  result: (jobId: string) => [...valKeys.all(), 'result', jobId] as const,
  logs: (jobId: string) => [...valKeys.all(), 'logs', jobId] as const,
  validate: (jobIds: string[]) => [...valKeys.all(), 'validate', jobIds] as const,
};
