// Generated from Rust by tools/generate_contracts.py. Do not edit.

export type Error = {
  "code": string;
  "message": string;
  [key: string]: unknown;
};

export type ErrorResponse = {
  "error": Error;
  "schema_version": number;
  [key: string]: unknown;
};
