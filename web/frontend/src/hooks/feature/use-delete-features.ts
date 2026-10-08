import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { featureApi } from '@/services/api/feature';
import { featureKeys } from './query-keys';
import type { Feature } from '@/types/feature';

export function useDeleteFeatures() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (featureIds: number[]) =>
      Promise.all(featureIds.map((id) => featureApi.deleteFeature(id))),
    onMutate: async (featureIds) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: featureKeys.lists() });
      await Promise.all(
        featureIds.map((id) => queryClient.cancelQueries({ queryKey: featureKeys.detail(id) }))
      );

      // Snapshot previous value for rollback
      const previousFeatures = queryClient.getQueryData<Feature[]>(featureKeys.list());

      // Optimistically remove features from cache
      queryClient.setQueryData<Feature[]>(featureKeys.list(), (old) =>
        old?.filter((feature) => !featureIds.includes(feature.id))
      );

      return { previousFeatures };
    },
    onError: (_error, _featureIds, context) => {
      // Rollback on error
      if (context?.previousFeatures) {
        queryClient.setQueryData(featureKeys.list(), context.previousFeatures);
      }
      toast.error('批量删除特征算子失败，请重试');
    },
    onSuccess: (_data, featureIds) => {
      toast.success(`成功删除 ${featureIds.length} 个特征算子`);
    },
    onSettled: (_data, _error, featureIds) => {
      // Always refetch after error or success to ensure cache is in sync
      queryClient.invalidateQueries({ queryKey: featureKeys.lists() });
      featureIds.forEach((id) => {
        queryClient.invalidateQueries({ queryKey: featureKeys.detail(id) });
      });
    },
  });
}
