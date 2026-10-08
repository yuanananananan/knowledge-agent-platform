import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { trainApi } from '@/services/api/train';
import { trainKeys } from './query-keys';
import type { TrainTaskData } from '@/types/task';

export function useDeleteTrainingTask() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => trainApi.deleteTrainingTask(jobId),
    onMutate: async (jobId) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: trainKeys.lists() });
      await queryClient.cancelQueries({ queryKey: trainKeys.logs(jobId) });
      await queryClient.cancelQueries({ queryKey: trainKeys.result(jobId) });

      // Snapshot previous value for rollback
      const previousTasks = queryClient.getQueryData<TrainTaskData[]>(trainKeys.lists());

      // Optimistically remove task from cache
      queryClient.setQueryData<TrainTaskData[]>(trainKeys.lists(), (old) =>
        old?.filter((task) => task.jobId !== jobId)
      );

      return { previousTasks };
    },
    onError: (_error, _jobId, context) => {
      // Rollback on error
      if (context?.previousTasks) {
        queryClient.setQueryData(trainKeys.lists(), context.previousTasks);
      }
      toast.error('删除训练任务失败，请重试');
    },
    onSuccess: () => {
      toast.success('训练任务删除成功');
    },
    onSettled: (_data, _error, jobId) => {
      // Always refetch after error or success to ensure cache is in sync
      queryClient.invalidateQueries({ queryKey: trainKeys.lists() });
      queryClient.invalidateQueries({ queryKey: trainKeys.logs(jobId) });
      queryClient.invalidateQueries({ queryKey: trainKeys.result(jobId) });
    },
  });
}
