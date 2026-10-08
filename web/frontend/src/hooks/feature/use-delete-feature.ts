import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { featureApi } from '@/services/api/feature';
import { featureKeys } from './query-keys';
import type { Feature } from '@/types/feature';

export function useDeleteFeature() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (featureId: number) => featureApi.deleteFeature(featureId),
    onMutate: async (featureId) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: featureKeys.lists() });
      await queryClient.cancelQueries({ queryKey: featureKeys.detail(featureId) });

      // Snapshot previous value for rollback
      const previousFeatures = queryClient.getQueryData<Feature[]>(featureKeys.list());

      // Optimistically remove feature from cache
      queryClient.setQueryData<Feature[]>(featureKeys.list(), (old) =>
        old?.filter((feature) => feature.id !== featureId)
      );

      return { previousFeatures };
    },
    onError: (_error, _featureId, context) => {
      // Rollback on error
      if (context?.previousFeatures) {
        queryClient.setQueryData(featureKeys.list(), context.previousFeatures);
      }
      toast.error('删除特征算子失败，请重试');
    },
    onSuccess: () => {
      toast.success('特征算子删除成功');
    },
    onSettled: (_data, _error, featureId) => {
      // Always refetch after error or success to ensure cache is in sync
      queryClient.invalidateQueries({ queryKey: featureKeys.lists() });
      queryClient.invalidateQueries({ queryKey: featureKeys.detail(featureId) });
    },
  });
}
