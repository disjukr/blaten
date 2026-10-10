import { f32, type Tensor } from 'blaten';
export type Image = Tensor<readonly [2, 3], typeof f32>;
