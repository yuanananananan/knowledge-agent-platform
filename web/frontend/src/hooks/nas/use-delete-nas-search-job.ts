import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { nasApi } from '@/services/api/nas';
import { nasKeys } from './query-keys';

export function useDeleteNasSearchJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => nasApi.deleteNasSearchJob(id),

    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: nasKeys.lists(),
      });

      toast.success('NAS搜索任务已删除');
    },

    onError: () => {
      toast.error('删除NAS搜索任务失败');
    },
  });
}