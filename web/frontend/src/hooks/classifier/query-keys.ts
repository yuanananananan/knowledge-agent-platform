export const classifierKeys = {
  all: () => ['classifier'] as const,
  list: () => [...classifierKeys.all(), 'list'] as const,
  detail: (id: number) => [...classifierKeys.all(), 'detail', id] as const,
};
