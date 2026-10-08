import { httpService } from '@/services/http';
import type { User, UserRole } from '@/types/auth';

export const userApi = {
  async getUserList() {
    return await httpService.get<User[]>('/api/user');
  },

  async createUser(data: { userAccount: string; userName: string; userPassword: string }) {
    return await httpService.post('/api/register', data);
  },

  async updateUser(
    id: number,
    data: { userName?: string; userPassword?: string; userRole?: UserRole }
  ) {
    return await httpService.put(`/api/user/${id}`, data);
  },

  async deleteUser(id: number) {
    return await httpService.delete(`/api/user/${id}`);
  },
};
