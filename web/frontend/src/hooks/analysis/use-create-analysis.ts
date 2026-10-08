import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { analysisApi, type CreateEffectivenessAnalysisRequest } from '@/services/api/analysis';
import { analysisKeys } from './query-keys';

export function useCreateEffectivenessAnalysis() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateEffectivenessAnalysisRequest) => analysisApi.createAnalysis(data),
    onSuccess: (record) => {
      queryClient.invalidateQueries({ queryKey: analysisKeys.lists(record.userId) });
      toast.success('有效性分析创建成功');
    },
    onError: () => {
      toast.error('有效性分析创建失败，请重试');
    },
  });
}