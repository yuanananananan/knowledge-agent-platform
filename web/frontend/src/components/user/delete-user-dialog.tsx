import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { useDeleteUser } from '@/hooks/user/use-delete-user';
import { useDeleteUsers } from '@/hooks/user/use-delete-users';
import { useAuthStore } from '@/stores/auth';
import type { User } from '@/types/auth';

interface DeleteUserDialogProps {
  open: boolean;
  onOpenChange: (_open: boolean) => void;
  user: User | null;
}

export function DeleteUserDialog({ open, onOpenChange, user }: DeleteUserDialogProps) {
  const { mutate: deleteUser, isPending } = useDeleteUser();
  const { user: currentUser } = useAuthStore();

  const handleDelete = async () => {
    if (!user) return;

    deleteUser(user.id, {
      onSuccess: () => {
        onOpenChange(false);
        // Logout logic is handled in useDeleteUser hook
      },
    });
  };

  const isOwnAccount = currentUser && user && currentUser.id === user.id;

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="text-base font-semibold">
            {isOwnAccount ? '注销账号' : '删除用户'}
          </AlertDialogTitle>
          <AlertDialogDescription>
            {isOwnAccount ? (
              <>
                确定要注销自己的账号吗？
                <strong>注销后将立即退出登录，且无法恢复此账号。</strong>
              </>
            ) : (
              <>
                确定要删除用户 <strong>{user?.userName || user?.userAccount}</strong> 吗？
              </>
            )}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isPending}>取消</AlertDialogCancel>
          <AlertDialogAction onClick={handleDelete} disabled={isPending}>
            {isPending ? '处理中...' : isOwnAccount ? '确认注销' : '确认删除'}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

// 批量删除对话框
interface DeleteUsersDialogProps {
  open: boolean;
  onOpenChange: (_open: boolean) => void;
  users: User[];
  onSuccess?: () => void;
}

export function DeleteUsersDialog({ open, onOpenChange, users }: DeleteUsersDialogProps) {
  const { mutate: deleteUsers, isPending } = useDeleteUsers();
  const { user: currentUser } = useAuthStore();

  const handleDelete = async () => {
    if (users.length === 0) return;

    const userIds = users.map((u) => u.id);
    deleteUsers(userIds, {
      onSuccess: () => {
        onOpenChange(false);
        // Logout logic is handled in useDeleteUsers hook
      },
    });
  };

  const includesSelf = currentUser && users.some((u) => u.id === currentUser.id);

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="text-base font-semibold">批量删除用户</AlertDialogTitle>
          <AlertDialogDescription>
            确定要删除选中的 <strong>{users.length}</strong> 个用户吗？
            <br />
            {includesSelf && (
              <span className="font-semibold">
                注意：选中的用户中包含你自己的账号，删除后将立即退出登录。
              </span>
            )}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isPending}>取消</AlertDialogCancel>
          <AlertDialogAction onClick={handleDelete} disabled={isPending || users.length === 0}>
            {isPending ? '删除中...' : '确认删除'}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
