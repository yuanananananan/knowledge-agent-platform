import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { userApi } from '@/services/api/user';
import { useAuthStore } from '@/stores/auth';
import { userKeys } from './query-keys';
import type { User, UserRole } from '@/types/auth';
import { authApi } from '@/services/api/auth';

interface UpdateUserVariables {
  id: number;
  updates: {
    userName?: string;
    userPassword?: string;
    userRole?: UserRole;
  };
}

export function useUpdateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, updates }: UpdateUserVariables) => userApi.updateUser(id, updates),
    onMutate: async ({ id, updates }) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: userKeys.lists() });

      // Snapshot previous value for rollback
      const previousUsers = queryClient.getQueryData<User[]>(userKeys.list());

      // Optimistically update cache (only for non-password changes)
      if (!updates.userPassword) {
        queryClient.setQueryData<User[]>(userKeys.list(), (old) =>
          old?.map((user) => (user.id === id ? { ...user, ...updates } : user))
        );
      }

      return { previousUsers };
    },
    onError: (_error, _variables, context) => {
      // Rollback on error
      if (context?.previousUsers) {
        queryClient.setQueryData(userKeys.list(), context.previousUsers);
      }
      toast.error('更新用户信息失败，请重试');
    },
    onSuccess: (_data, variables) => {
      const { user: currentUser, resetAuthState } = useAuthStore.getState();
      const isChangingOwnPassword =
        variables.id === currentUser?.id && variables.updates.userPassword;

      if (isChangingOwnPassword) {
        toast.success('密码修改成功，请重新登录');
        setTimeout(() => {
          authApi.logout();
          resetAuthState();
          // Navigation to login will be handled by auth guard
        }, 1000);
      } else {
        toast.success('用户信息更新成功');
      }
    },
    onSettled: (_data, _error, { id }) => {
      // Always refetch after error or success to ensure cache is in sync
      queryClient.invalidateQueries({ queryKey: userKeys.lists() });
      queryClient.invalidateQueries({ queryKey: userKeys.detail(id) });
    },
  });
}
