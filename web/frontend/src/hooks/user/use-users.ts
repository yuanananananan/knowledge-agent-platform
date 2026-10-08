import { useQuery } from '@tanstack/react-query';
import { userApi } from '@/services/api/user';
import { useAuthStore } from '@/stores/auth';
import { userKeys } from './query-keys';

export function useUsers() {
  const { user: currentUser } = useAuthStore();

  return useQuery({
    queryKey: userKeys.list(),
    queryFn: userApi.getUserList,
    select: (data) => {
      // Admin can see all users, regular users can only see themselves
      if (currentUser?.userRole === 'admin') {
        return data;
      }
      return data.filter((user) => user.id === currentUser?.id);
    },
    enabled: !!currentUser, // Only fetch when user is authenticated
  });
}
