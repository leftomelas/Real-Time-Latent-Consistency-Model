declare module 'piexifjs' {
  export const ImageIFD: {
    Make: number;
    ImageDescription: number;
    Software: number;
  };
  export const ExifIFD: {
    DateTimeOriginal: number;
  };
  export function dump(exifObj: any): any;
  export function insert(exifBytes: any, dataURL: string): string;
}
