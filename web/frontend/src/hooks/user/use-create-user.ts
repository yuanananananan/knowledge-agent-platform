import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { userApi } from '@/services/api/user';
import { userKeys } from './query-keys';

interface CreateUserVariables {
  userAccount: string;
  userName: string;
  userPassword: string;
}

export function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateUserVariables) => userApi.createUser(data),
    onSuccess: () => {
      // Invalidate all user lists
      queryClient.invalidateQueries({ queryKey: userKeys.lists() });
      toast.success('用户创建成功');
    },
    onError: (_error) => {
      toast.error('创建用户失败，请重试');
    },
  });
}
