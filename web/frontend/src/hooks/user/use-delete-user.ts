import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { userApi } from '@/services/api/user';
import { useAuthStore } from '@/stores/auth';
import { userKeys } from './query-keys';
import type { User } from '@/types/auth';
import { authApi } from '@/services/api/auth';

export function useDeleteUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (userId: number) => userApi.deleteUser(userId),
    onMutate: async (userId) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: userKeys.lists() });
      await queryClient.cancelQueries({ queryKey: userKeys.detail(userId) });

      // Snapshot previous value for rollback
      const previousUsers = queryClient.getQueryData<User[]>(userKeys.list());

      // Optimistically remove user from cache
      queryClient.setQueryData<User[]>(userKeys.list(), (old) =>
        old?.filter((user) => user.id !== userId)
      );

      return { previousUsers };
    },
    onError: (_error, _userId, context) => {
      // Rollback on error
      if (context?.previousUsers) {
        queryClient.setQueryData(userKeys.list(), context.previousUsers);
      }
      toast.error('删除用户失败，请重试');
    },
    onSuccess: (_data, userId) => {
      const { user: currentUser, resetAuthState } = useAuthStore.getState();

      if (userId === currentUser?.id) {
        toast.warning('账号已注销，即将退出登录');
        setTimeout(() => {
          authApi.logout();
          resetAuthState();
          // Navigation to login will be handled by auth guard
        }, 1000);
      } else {
        toast.success('用户删除成功');
      }
    },
    onSettled: (_data, _error, userId) => {
      // Always refetch after error or success to ensure cache is in sync
      queryClient.invalidateQueries({ queryKey: userKeys.lists() });
      queryClient.invalidateQueries({ queryKey: userKeys.detail(userId) });
    },
  });
}
