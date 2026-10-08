export const userKeys = {
  all: () => ['user'] as const,
  lists: () => [...userKeys.all(), 'list'] as const,
  list: (filters?: { role?: string }) => [...userKeys.lists(), filters] as const,
  detail: (id: number) => [...userKeys.all(), 'detail', id] as const,
};
