export interface DatasetMetadata {
  readonly name: string;
  readonly split: string;
  readonly homepage?: string;
  readonly classNames?: readonly string[];
}

/** Indexed samples, decoded on demand. Loading a dataset does not allocate tensors. */
export interface Dataset<Sample, Metadata extends DatasetMetadata = DatasetMetadata> {
  readonly count: number;
  readonly metadata: Metadata;
  get(index: number): Promise<Sample>;
}

/** Interleaved Hwc bytes. Dataset views are borrowed; copy before editing. */
export interface Image<Features extends number = number> {
  readonly data: Uint8Array;
  readonly width: number;
  readonly height: number;
  readonly channels: number;
  readonly featureCount: Features;
}
export interface ImageClassificationSample<Features extends number = number> {
  readonly image: Image<Features>;
  readonly label: number;
}
export interface ImageClassificationMetadata<Classes extends number = number> extends DatasetMetadata {
  readonly classNames: readonly string[] & { readonly length: Classes };
}
