import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { nasApi } from '@/services/api/nas';
import type { StartNasSearchRequest } from '@/types/nas';
import { nasKeys } from './query-keys';

export function useStartNasSearch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: StartNasSearchRequest) => nasApi.startNasSearch(data),

    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: nasKeys.lists(),
      });

      toast.success('NAS主干搜索任务已启动');
    },

    onError: () => {
      toast.error('启动NAS主干搜索失败');
    },
  });
}