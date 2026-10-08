import { useState } from 'react';
import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnFiltersState,
  type SortingState,
  type VisibilityState,
} from '@tanstack/react-table';
import { ChevronDown } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useAuthStore } from '@/stores/auth';
import { createColumns } from '@/components/user/user-columns';
import { AddUserModal } from '@/components/user/add-user-modal';
import { PasswordModal } from '@/components/user/password-modal';
import { DeleteUserDialog, DeleteUsersDialog } from '@/components/user/delete-user-dialog';
import { useUsers } from '@/hooks/user/use-users';
import type { User } from '@/types/auth';

export default function UserManagement() {
  const { user: currentUser } = useAuthStore();
  const isAdmin = currentUser?.userRole === 'admin';
  const { data: users = [], isLoading } = useUsers();

  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});
  const [rowSelection, setRowSelection] = useState({});

  // Modal states
  const [addUserModalOpen, setAddUserModalOpen] = useState(false);
  const [passwordModalOpen, setPasswordModalOpen] = useState(false);
  const [deleteUserModalOpen, setDeleteUserModalOpen] = useState(false);
  const [deleteUsersModalOpen, setDeleteUsersModalOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);

  const columns = createColumns({
    onPasswordClick: (user: User) => {
      setSelectedUser(user);
      setPasswordModalOpen(true);
    },
    onDeleteClick: (user: User) => {
      setSelectedUser(user);
      setDeleteUserModalOpen(true);
    },
  });

  const table = useReactTable({
    data: users,
    columns,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    initialState: {
      pagination: {
        pageSize: 10,
      },
    },
    state: {
      sorting,
      columnFilters,
      columnVisibility,
      rowSelection,
    },
  });

  const selectedRows = table.getFilteredSelectedRowModel().rows;
  const selectedUsers = selectedRows.map((row) => row.original);

  return (
    <div className="container mx-auto max-w-7xl p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <h1 className="text-xl font-semibold">用户管理</h1>
          <p className="text-sm sm:text-base text-muted-foreground">管理用户账号和权限</p>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Input
            placeholder="搜索用户名..."
            value={(table.getColumn('userAccount')?.getFilterValue() as string) ?? ''}
            onChange={(event) => table.getColumn('userAccount')?.setFilterValue(event.target.value)}
            className="max-w-sm"
          />
          <Select
            value={(table.getColumn('userRole')?.getFilterValue() as string) ?? 'all'}
            onValueChange={(value: string) => {
              table.getColumn('userRole')?.setFilterValue(value === 'all' ? '' : value);
            }}
          >
            <SelectTrigger className="w-32">
              <SelectValue placeholder="角色筛选" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部角色</SelectItem>
              <SelectItem value="admin">管理员</SelectItem>
              <SelectItem value="user">用户</SelectItem>
              <SelectItem value="ban">禁用</SelectItem>
            </SelectContent>
          </Select>
          {isAdmin && selectedRows.length > 0 && (
            <Button variant="secondary" size="sm" onClick={() => setDeleteUsersModalOpen(true)}>
              删除选中 ({selectedRows.length})
            </Button>
          )}
        </div>
        <div className="flex items-center space-x-2">
          {isAdmin && <Button onClick={() => setAddUserModalOpen(true)}>添加用户</Button>}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                列显示 <ChevronDown className="ml-2 h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {table
                .getAllColumns()
                .filter((column) => column.getCanHide())
                .map((column) => {
                  return (
                    <DropdownMenuCheckboxItem
                      key={column.id}
                      className="capitalize"
                      checked={column.getIsVisible()}
                      onCheckedChange={(value) => column.toggleVisibility(!!value)}
                    >
                      {column.id === 'id' && 'ID'}
                      {column.id === 'userAccount' && '用户名'}
                      {column.id === 'userName' && '姓名'}
                      {column.id === 'userRole' && '角色'}
                      {column.id === 'actions' && '操作'}
                    </DropdownMenuCheckboxItem>
                  );
                })}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div className="rounded-md border">
        <Table className="table-fixed w-full">
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  return (
                    <TableHead key={header.id} style={{ width: header.column.getSize() }}>
                      {header.isPlaceholder
                        ? null
                        : flexRender(header.column.columnDef.header, header.getContext())}
                    </TableHead>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-24 text-center">
                  加载中...
                </TableCell>
              </TableRow>
            ) : table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} data-state={row.getIsSelected() && 'selected'}>
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id} style={{ width: cell.column.getSize() }}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-24 text-center">
                  没有找到用户数据
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      <div className="flex items-center justify-between space-x-2 py-4">
        <div className="text-muted-foreground flex-1 text-sm">
          {table.getFilteredSelectedRowModel().rows.length} /{' '}
          {table.getFilteredRowModel().rows.length} 行已选择
        </div>
        <div className="flex items-center space-x-2">
          <p className="text-sm font-medium">每页显示</p>
          <Select
            value={`${table.getState().pagination.pageSize}`}
            onValueChange={(value: string) => {
              table.setPageSize(Number(value));
            }}
          >
            <SelectTrigger className="h-8 w-[70px]">
              <SelectValue placeholder={table.getState().pagination.pageSize} />
            </SelectTrigger>
            <SelectContent side="top">
              {[10, 20, 50].map((pageSize) => (
                <SelectItem key={pageSize} value={`${pageSize}`}>
                  {pageSize}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            上一页
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            下一页
          </Button>
        </div>
      </div>

      {/* Modals */}
      <AddUserModal open={addUserModalOpen} onOpenChange={setAddUserModalOpen} />
      <PasswordModal
        open={passwordModalOpen}
        onOpenChange={setPasswordModalOpen}
        user={selectedUser}
      />
      <DeleteUserDialog
        open={deleteUserModalOpen}
        onOpenChange={setDeleteUserModalOpen}
        user={selectedUser}
      />
      <DeleteUsersDialog
        open={deleteUsersModalOpen}
        onOpenChange={setDeleteUsersModalOpen}
        users={selectedUsers}
        onSuccess={() => {
          setRowSelection({});
        }}
      />
    </div>
  );
}
