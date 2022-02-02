import { ApiHookResult } from './types';
import axios, { AxiosResponse } from 'axios';
import { useQuery } from 'react-query';

export type Position = {
  Symbol: string;
  Price: number;
  Quantity: number;
  time: Date;
};

export type PositionsResponse = {
  positions: Position[];
};

export function usePositions(): ApiHookResult<PositionsResponse> {
  const fetch = () => axios.get('/exchange/positions');
  const {
    data,
    isLoading: loading,
    error,
  } = useQuery<AxiosResponse<PositionsResponse, Error>, Error>(
    'positions',
    fetch,
    { refetchInterval: 4 * 1000 }
  );

  return { data: data?.data, loading, error };
}