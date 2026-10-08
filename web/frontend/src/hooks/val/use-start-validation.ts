import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { valApi } from '@/services/api/val';
import { valKeys } from './query-keys';

interface StartValidationVariables {
  userId: number;
  trainJobId: string;
}

export function useStartValidation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: StartValidationVariables) => valApi.startValidation(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: valKeys.lists() });
      toast.success('测试任务已启动');
    },
    onError: () => {
      toast.error('启动测试失败，请重试');
    },
  });
}
