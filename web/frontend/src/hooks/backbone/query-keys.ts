export const backboneKeys = {
  all: () => ['backbone'] as const,
  list: () => [...backboneKeys.all(), 'list'] as const,
  detail: (id: number) => [...backboneKeys.all(), 'detail', id] as const,
};
