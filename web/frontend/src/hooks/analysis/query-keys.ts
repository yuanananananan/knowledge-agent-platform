export const analysisKeys = {
  all: () => ['analysis'] as const,
  lists: (userId: number) => [...analysisKeys.all(), 'list', userId] as const,
  detail: (analysisId: string) => [...analysisKeys.all(), 'detail', analysisId] as const,
};