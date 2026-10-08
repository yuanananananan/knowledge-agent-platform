import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { userApi } from '@/services/api/user';
import { useAuthStore } from '@/stores/auth';
import { userKeys } from './query-keys';
import type { User } from '@/types/auth';
import { authApi } from '@/services/api/auth';

export function useDeleteUsers() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (userIds: number[]) => Promise.all(userIds.map((id) => userApi.deleteUser(id))),
    onMutate: async (userIds) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: userKeys.lists() });
      await Promise.all(
        userIds.map((id) => queryClient.cancelQueries({ queryKey: userKeys.detail(id) }))
      );

      // Snapshot previous value for rollback
      const previousUsers = queryClient.getQueryData<User[]>(userKeys.list());

      // Optimistically remove users from cache
      queryClient.setQueryData<User[]>(userKeys.list(), (old) =>
        old?.filter((user) => !userIds.includes(user.id))
      );

      return { previousUsers };
    },
    onError: (_error, _userIds, context) => {
      // Rollback on error
      if (context?.previousUsers) {
        queryClient.setQueryData(userKeys.list(), context.previousUsers);
      }
      toast.error('批量删除用户失败，请重试');
    },
    onSuccess: (_data, userIds) => {
      const { user: currentUser, resetAuthState } = useAuthStore.getState();
      const deletedSelf = userIds.includes(currentUser?.id || 0);

      if (deletedSelf) {
        toast.warning('您的账号已被删除，即将退出登录');
        setTimeout(() => {
          authApi.logout();
          resetAuthState();
          // Navigation to login will be handled by auth guard
        }, 1000);
      } else {
        toast.success(`成功删除 ${userIds.length} 个用户`);
      }
    },
    onSettled: (_data, _error, userIds) => {
      // Always refetch after error or success to ensure cache is in sync
      queryClient.invalidateQueries({ queryKey: userKeys.lists() });
      userIds.forEach((id) => {
        queryClient.invalidateQueries({ queryKey: userKeys.detail(id) });
      });
    },
  });
}
