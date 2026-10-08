import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { valApi } from '@/services/api/val';
import { valKeys } from './query-keys';

export function useDeleteValJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (jobId: string) => {
      await valApi.deleteValJob(jobId);
      return jobId;
    },
    onSuccess: async (deletedJobId) => {
      // 先直接更新列表缓存，页面立即变化
      queryClient.setQueryData(valKeys.lists(), (oldData: any) => {
        if (!Array.isArray(oldData)) return oldData;
        return oldData.filter((item) => item.valJobId !== deletedJobId);
      });

      // 再让列表重新请求一次，和后端保持一致
      await queryClient.invalidateQueries({ queryKey: valKeys.lists() });
      await queryClient.refetchQueries({ queryKey: valKeys.lists(), type: 'active' });

      toast.success('测试任务删除成功');
    },
    onError: () => {
      toast.error('测试任务删除失败');
    },
  });
}