import { httpService } from '@/services/http';

export const authApi = {
  async login(data: { userAccount: string; userPassword: string }) {
    const response = await httpService.post<{
      message?: string;
      user?: {
        id?: number;
        userAccount?: string;
        userName?: string;
        userRole?: string;
      };
    }>('/api/user/login', data);
    const user = {
      id: response.user?.id ?? 0,
      userAccount: response.user?.userAccount ?? '',
      userName: response.user?.userName ?? '',
      userRole: response.user?.userRole ?? '',
    };
    return user;
  },

  async register(data: { userAccount: string; userName: string; userPassword: string }) {
    return await httpService.post('/api/user/register', data);
  },

  async logout() {
    await httpService.post('/api/user/logout', undefined);
  },
};
