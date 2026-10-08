import type { ColumnDef } from '@tanstack/react-table';
import { MoreHorizontal, Trash2, Key } from 'lucide-react';
import { Checkbox } from '@/components/ui/checkbox';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useAuthStore } from '@/stores/auth';
import type { User } from '@/types/auth';
import EditableUserName from '@/components/user/editable-user-name';
import RoleSelect from '@/components/user/role-select';

interface ColumnCallbacks {
  onPasswordClick: (user: User) => void;
  onDeleteClick: (user: User) => void;
}

export const createColumns = (callbacks: ColumnCallbacks): ColumnDef<User>[] => [
  {
    id: 'select',
    header: ({ table }) => {
      const currentUser = useAuthStore.getState().user;
      const isAdmin = currentUser?.userRole === 'admin';

      if (!isAdmin) return null;

      return (
        <Checkbox
          checked={
            table.getIsAllPageRowsSelected() ||
            (table.getIsSomePageRowsSelected() && 'indeterminate')
          }
          onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
          aria-label="Select all"
        />
      );
    },
    cell: ({ row }) => {
      const currentUser = useAuthStore.getState().user;
      const isAdmin = currentUser?.userRole === 'admin';

      if (!isAdmin) return null;

      return (
        <Checkbox
          checked={row.getIsSelected()}
          onCheckedChange={(value) => row.toggleSelected(!!value)}
          aria-label="Select row"
        />
      );
    },
    enableSorting: false,
    enableHiding: false,
    size: 50,
  },
  {
    accessorKey: 'id',
    header: 'ID',
    cell: ({ row }) => <div className="w-[60px]">{row.getValue('id')}</div>,
    size: 80,
  },
  {
    accessorKey: 'userAccount',
    header: '用户名',
    cell: ({ row }) => {
      const userAccount = row.getValue('userAccount') as string;
      return <div className="font-medium w-32">{userAccount}</div>;
    },
    size: 100,
  },
  {
    accessorKey: 'userName',
    header: '姓名',
    cell: ({ row }) => <EditableUserName user={row.original} />,
    size: 100,
  },
  {
    accessorKey: 'userRole',
    header: '角色',
    cell: ({ row }) => {
      const currentUser = useAuthStore.getState().user;
      const isAdmin = currentUser?.userRole === 'admin';
      return isAdmin ? (
        <RoleSelect user={row.original} />
      ) : (
        <span className="text-sm text-muted-foreground">
          {row.original.userRole === 'admin'
            ? '管理员'
            : row.original.userRole === 'user'
              ? '用户'
              : '禁用'}
        </span>
      );
    },
    filterFn: (row, id, value) => {
      return value.includes(row.getValue(id));
    },
    size: 100,
  },
  {
    id: 'actions',
    header: '操作',
    cell: ({ row }) => {
      const user = row.original;
      const currentUser = useAuthStore.getState().user;
      const isAdmin = currentUser?.userRole === 'admin';
      const canEdit = isAdmin || user.id === currentUser?.id;

      if (!canEdit) {
        return null;
      }

      return (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-8 w-8 p-0">
              <span className="sr-only">打开菜单</span>
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              onClick={() => {
                callbacks.onPasswordClick(user);
              }}
            >
              <Key className="mr-2 h-4 w-4" />
              修改密码
            </DropdownMenuItem>
            {isAdmin && (
              <DropdownMenuItem
                onClick={() => {
                  callbacks.onDeleteClick(user);
                }}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                删除用户
              </DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      );
    },
    size: 80,
  },
];
