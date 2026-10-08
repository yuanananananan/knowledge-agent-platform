import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useUpdateUser } from '@/hooks/user/use-update-user';
import { useAuthStore } from '@/stores/auth';
import type { User } from '@/types/auth';

const passwordSchema = z
  .object({
    newPassword: z.string().min(6, {
      message: '密码至少需要6位',
    }),
    confirmPassword: z.string().min(1, {
      message: '请确认密码',
    }),
  })
  .refine((data) => data.newPassword === data.confirmPassword, {
    message: '两次输入的密码不一致',
    path: ['confirmPassword'],
  });

interface PasswordModalProps {
  open: boolean;
  onOpenChange: (_open: boolean) => void;
  user: User | null;
}

export function PasswordModal({ open, onOpenChange, user }: PasswordModalProps) {
  const { mutate: updateUser, isPending } = useUpdateUser();
  const { user: currentUser } = useAuthStore();

  const form = useForm<z.infer<typeof passwordSchema>>({
    resolver: zodResolver(passwordSchema),
    defaultValues: {
      newPassword: '',
      confirmPassword: '',
    },
  });

  const onSubmit = async (values: z.infer<typeof passwordSchema>) => {
    if (!user) return;

    updateUser(
      {
        id: user.id,
        updates: {
          userPassword: values.newPassword,
        },
      },
      {
        onSuccess: () => {
          form.reset();
          onOpenChange(false);
          // Logout logic is handled in useUpdateUser hook
        },
      }
    );
  };

  const handleClose = () => {
    form.reset();
    onOpenChange(false);
  };

  const isOwnPassword = currentUser && user && currentUser.id === user.id;

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle className="text-base font-semibold">
            {isOwnPassword ? '修改我的密码' : `修改用户密码 - ${user?.userName}`}
          </DialogTitle>
        </DialogHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="newPassword"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>新密码</FormLabel>
                  <FormControl>
                    <Input
                      type="password"
                      placeholder="请输入新密码（至少6位）"
                      {...field}
                      disabled={isPending}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="confirmPassword"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>确认密码</FormLabel>
                  <FormControl>
                    <Input
                      type="password"
                      placeholder="请再次输入新密码"
                      {...field}
                      disabled={isPending}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <DialogFooter>
              <Button type="button" variant="outline" onClick={handleClose} disabled={isPending}>
                取消
              </Button>
              <Button type="submit" disabled={isPending}>
                {isPending ? '修改中...' : '确认修改'}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
