export const featureKeys = {
  all: () => ['feature'] as const,
  lists: () => [...featureKeys.all(), 'list'] as const,
  list: (filters?: { modality?: string }) => [...featureKeys.lists(), filters] as const,
  detail: (id: number) => [...featureKeys.all(), 'detail', id] as const,
};
