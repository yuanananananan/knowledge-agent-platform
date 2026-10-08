export const taskKeys = {
  all: () => ['task'] as const,
  extraConfigs: () => [...taskKeys.all(), 'extra-config'] as const,
};
