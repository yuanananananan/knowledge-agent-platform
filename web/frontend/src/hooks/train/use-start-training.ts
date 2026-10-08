import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { trainApi, type StartTrainingRequest } from '@/services/api/train';
import { trainKeys } from './query-keys';

export function useStartTraining() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: StartTrainingRequest) => trainApi.startTraining(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: trainKeys.lists() });
      toast.success('训练任务已启动');
    },
    onError: () => {
      toast.error('启动训练失败，请重试');
    },
  });
}
