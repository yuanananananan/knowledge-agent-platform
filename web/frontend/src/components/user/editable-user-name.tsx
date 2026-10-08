import { useAuthStore } from '@/stores/auth';
import type { User } from '@/types/auth';
import { useState } from 'react';
import { Input } from '../ui/input';
import { useUpdateUser } from '@/hooks/user/use-update-user';

// 内联编辑昵称组件
export default function EditableUserName({ user }: { user: User }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(user.userName || '');
  const { mutate: updateUser } = useUpdateUser();
  const currentUser = useAuthStore((state) => state.user);
  const isAdmin = currentUser?.userRole === 'admin';
  const canEdit = isAdmin || user.id === currentUser?.id;

  const handleSave = () => {
    if (editValue.trim() !== user.userName) {
      updateUser({
        id: user.id,
        updates: { userName: editValue.trim() },
      });
    }
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditValue(user.userName || '');
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      handleCancel();
    }
  };

  if (isEditing) {
    return (
      <div className="w-full">
        <Input
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onBlur={handleSave}
          onKeyDown={handleKeyDown}
          className="h-8 text-sm w-full"
          autoFocus
        />
      </div>
    );
  }

  if (!canEdit) {
    return <div className="px-2 py-1 min-h-6 flex items-center w-full">{user.userName || '-'}</div>;
  }

  return (
    <div
      className="cursor-pointer hover:bg-muted/50 px-2 py-1 rounded min-h-6 flex items-center w-full"
      onClick={() => setIsEditing(true)}
      title="点击编辑姓名"
    >
      {user.userName || '-'}
    </div>
  );
}
