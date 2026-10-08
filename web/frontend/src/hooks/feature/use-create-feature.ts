import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { featureApi } from '@/services/api/feature';
import { featureKeys } from './query-keys';
import type { Modality } from '@/types/dataset';

interface CreateFeatureVariables {
  name: string;
  modality: Modality;
  description: string;
  paramsSchema: string;
}

export function useCreateFeature() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateFeatureVariables) => featureApi.createFeature(data),
    onSuccess: () => {
      // Invalidate all feature lists
      queryClient.invalidateQueries({ queryKey: featureKeys.lists() });
      toast.success('特征算子创建成功');
    },
    onError: (_error) => {
      toast.error('创建特征算子失败，请重试');
    },
  });
}
