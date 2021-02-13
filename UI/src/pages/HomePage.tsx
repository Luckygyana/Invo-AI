import {
	Button,
	Typography,
	Grid,
	Card,
	CardContent,
	CardHeader,
	LinearProgress,
	Box,
	Modal,
} from "@material-ui/core";
import { makeStyles } from "@material-ui/styles";
import * as React from "react";

import { DropzoneArea } from "material-ui-dropzone";

import { history } from "../configureStore";
import {
	AttachFile,
	Description,
	PictureAsPdf,
	Theaters,
	Label,
} from "@material-ui/icons";
import Axios from "axios";
import { BASE_URL } from "../constants";

const handlePreviewIcon = (fileObject: any, classes: any) => {
	const { type } = fileObject.file;
	const iconProps = {
		className: classes.image,
	};

	if (type.startsWith("video/")) return <Theaters {...iconProps} />;

	switch (type) {
		case "application/msword":
		case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
			return <Description {...iconProps} />;
		case "application/pdf":
			return <PictureAsPdf {...iconProps} />;
		default:
			return <AttachFile {...iconProps} />;
	}
};

export function HomePage() {
	const classes = useStyles();
	const [boxColor, setBoxColor] = React.useState("red");
	const [files, setFiles] = React.useState<File[]>([]);

	const [uploading, setUploading] = React.useState(false);

	const [file, setFile] = React.useState<File>();

	const onButtonClick = () =>
		setBoxColor(boxColor === "red" ? "blue" : "red");

	const handleChange = (files: File[]) => {
		setFiles(files);
	};

	const onFileSelect = (event: any) => {
		console.log(event.target.files);
		setFile(event.target.files[0]);
	};

	const uplaodFile = async (file: File) => {
		setUploading(true);
		var formData = new FormData();
		formData.append("file", file);
		let res = await Axios.post(BASE_URL + "upload", formData, {
			headers: {
				"Content-Type": "multipart/form-data",
			},
		});
		console.log(res.data);
		setUploading(false);
		history.push("/job/" + res.data.jobId);
	};

	const uplaodFiles = async () => {
		setUploading(true);
		var formData = new FormData();
		files.forEach((f) => formData.append("files", f));
		let res = await Axios.post(BASE_URL + "upload", formData, {
			headers: {
				"Content-Type": "multipart/form-data",
			},
		});
		console.log(res.data);
		setUploading(false);
		history.push("/job/" + res.data.jobId);
	};


	return (
		<div className={classes.root}>
			<Typography variant="h6" gutterBottom>
				CREATE DIGITAL INVOICE
			</Typography>
			{/* {!file && (
				<Button
					variant="contained"
					component="label"
					style={{ backgroundColor: "#F8E831" }}
				>
					Select Invoice PDF
					<input
						type="file"
						style={{ display: "none" }}
						onChange={onFileSelect}
					/>
				</Button>
			)}

			{file && (
				<Card className={classes.filesContainer}>
					<CardHeader title={"Selected File : " + file.name} />
					<CardContent>
						<Grid container spacing={3} alignItems="center">
							<Grid item xs={2}>
								<Button
									style={{ backgroundColor: "#F8E831" }}
									onClick={() => uplaodFile(file)}
								>
									UPLOAD INVOICE
								</Button>
							</Grid>
							<Grid item xs={2}>
								<Button
									style={{ backgroundColor: "#ff5e50" }}
									onClick={() => {
										setFile(undefined);
									}}
								>
									Clear Selection
								</Button>
							</Grid>
						</Grid>
					</CardContent>
				</Card>
			)} */}

			<div className={classes.centerContainer}>
				<DropzoneArea
					onChange={handleChange}
					getPreviewIcon={handlePreviewIcon}
					filesLimit={100}
					dropzoneText="Click here to upload PDFs or drag and drop them."
					showPreviewsInDropzone={false}
				/>
			</div>

			{files && files.length > 0 && (
				<Card className={classes.filesContainer}>
					<CardHeader title="Selected Files" />
					<CardContent>
						<Grid container spacing={3}>
							{files.map((f) => (
								<Grid item xs={12}>
									<Typography variant="h6">
										{f.name}
									</Typography>
								</Grid>
							))}
							<Grid container spacing={3} alignItems="center">
								{!uploading && (
									<Grid item xs={12}>
										<Box m={2}>
											<Button
												style={{
													backgroundColor: "#F8E831",
												}}
												onClick={() => uplaodFiles()}
												disabled={files.length === 0}
											>
												UPLOAD INVOICES
											</Button>
										</Box>
									</Grid>
								)}
								{uploading && (
									<Grid item xs={12}>
										<Box m={2}>
											<Typography variant="h6">
												Uploading...
											</Typography>
										</Box>
									</Grid>
								)}
								{uploading && (
									<Grid item xs={12}>
										<LinearProgress />
									</Grid>
								)}

								{/* <Grid item xs={2}>
									<Button
										style={{ backgroundColor: "#ff5e50" }}
										onClick={() => {
											setFiles([]);
										}}
										disabled={files.length === 0}
									>
										Clear Selection
									</Button>
								</Grid> */}
							</Grid>
						</Grid>
					</CardContent>
				</Card>
			)}
		</div>
	);
}

const useStyles = makeStyles({
	root: {
		height: "100%",
		paddingTop: 20,
		paddingLeft: 15,
		paddingRight: 15,
	},

	centerContainer: {
		flex: 1,
		display: "flex",
		alignItems: "center",
		justifyContent: "center",
		flexDirection: "column",
	},

	button: {
		marginTop: 20,
		color: "yellow",
	},

	filesContainer: {
		marginTop: 20,
	},
});
