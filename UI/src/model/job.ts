export interface Job {
    _id: any,
    status: string,
    files: string[],
    output: any,
    existingInvoices: any,
    date_modified: number
}