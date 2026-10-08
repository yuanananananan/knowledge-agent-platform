export const nasKeys = {
  all: () => ['nas'] as const,
  lists: () => [...nasKeys.all(), 'list'] as const,
  logs: (jobId: string) => [...nasKeys.all(), 'logs', jobId] as const,
};