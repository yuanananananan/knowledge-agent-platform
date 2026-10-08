import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { User } from '@/types/auth';

interface AuthState {
  user: User;
  isAuthenticated: boolean;
  setUser: (value: User) => void;
  setAuthenticated: (value: boolean) => void;
  resetAuthState: () => void;
}

export const useAuthStore = create<AuthState>()(
  devtools(
    (set) => ({
      user: {
        id: 0,
        userName: '',
        userAccount: '',
        userRole: '',
      },
      isAuthenticated: false,
      setUser: (value: User) => set({ user: value }),
      setAuthenticated: (value: boolean) => set({ isAuthenticated: value }),
      resetAuthState: () =>
        set({
          user: {
            id: 0,
            userName: '',
            userAccount: '',
            userRole: '',
          },
          isAuthenticated: false,
        }),
    }),
    { name: 'AuthStore' }
  )
);
