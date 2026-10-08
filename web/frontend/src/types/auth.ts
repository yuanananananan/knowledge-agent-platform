export type User = {
  id: number;
  userName: string;
  userAccount: string;
  userRole: string;
};

export type UserRole = 'admin' | 'user' | 'ban';
