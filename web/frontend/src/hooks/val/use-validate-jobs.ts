import { useMutation } from '@tanstack/react-query';
import { toast } from 'sonner';
import { valApi } from '@/services/api/val';

interface ValidateJobsVariables {
  jobIds: string[];
}

export function useValidateJobs() {
  return useMutation({
    mutationFn: (data: ValidateJobsVariables) => valApi.validateJobs(data),
    onSuccess: () => {
      toast.success('有效性评估完成');
    },
    onError: () => {
      toast.error('有效性评估失败，请重试');
    },
  });
}
