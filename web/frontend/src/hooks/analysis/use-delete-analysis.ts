import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { analysisApi } from '@/services/api/analysis';
import { analysisKeys } from './query-keys';

export function useDeleteAnalysis(userId: number) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (analysisId: string) => analysisApi.deleteAnalysis(analysisId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: analysisKeys.lists(userId) });
      toast.success('分析记录删除成功');
    },
    onError: () => {
      toast.error('删除分析记录失败，请重试');
    },
  });
}