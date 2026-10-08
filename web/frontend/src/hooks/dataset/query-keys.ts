export const datasetKeys = {
  all: () => ['dataset'] as const,
  list: () => [...datasetKeys.all(), 'list'] as const,
  detail: (id: number) => [...datasetKeys.all(), 'detail', id] as const,
  images: (datasetId: number) => [...datasetKeys.all(), 'images', datasetId] as const,
};
