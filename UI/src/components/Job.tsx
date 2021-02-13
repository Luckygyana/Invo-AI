import * as React from "react";
import { makeStyles } from "@material-ui/styles";
import { Job } from "../model";
import {
	Paper,
	Card,
	CardHeader,
	CardContent,
	Grid,
	List,
	ListItem,
	ListItemText,
	Divider,
	Typography,
	Box,
	Button,
	GridList,
	GridListTile,
	GridListTileBar,
	Modal,
} from "@material-ui/core";
import Axios from "axios";
import { BASE_URL } from "../constants";

export function JobView(props: { jobId: string }) {
	const classes = useStyles();

	const [loading, setLoading] = React.useState(false);

	const [job, setJob] = React.useState<Job>();

	React.useEffect(() => {
		getJob();
	}, []);

	const getJob = async () => {
		setLoading(true);
		let res = await Axios.get(BASE_URL + "job/" + props.jobId);
		const data = res.data;
		setJob(data);
		setLoading(false);
	};
	const [open, setOpen] = React.useState(false);

	const [image, setImage] = React.useState("");

	const [xlsx, setXlsx] = React.useState("");

	return (
		<>
			<div style={{ height: "20px" }}>{loading ? "Loading..." : ""}</div>
			<Modal
				open={open}
				onClose={() => {
					setOpen(false);
				}}
			>
				<Grid
					container
					alignContent="center"
					alignItems="center"
					justify="center"
				>
					<Grid item xs={12}>
						<img
							style={{ height: "750px", margin: "auto" }}
							src={image}
						/>
					</Grid>
					<Grid item xs={12}>
						<Box ml={24} mb={2}>
							<Button
								variant="contained"
								style={{
									backgroundColor: "#F8E831",
								}}
								onClick={() => {
									window.open(xlsx, "_blank");
								}}
							>
								Process
							</Button>
						</Box>
					</Grid>
				</Grid>
			</Modal>
			<Paper>
				{job && (
					<Card>
						<CardHeader title={"ID : " + props.jobId} />
						<CardContent>
							<Grid container>
								<Grid item xs={12}>
									<Typography variant="h6">
										{"Number of Invoices : " +
											job.files.length}
									</Typography>
								</Grid>
							</Grid>
							<Box m={2}></Box>
							<Divider />
							<Box m={2}></Box>
							{Object.keys(job.output).map((invoice, i) => {
								let exist = false;
								console.log(job.existingInvoices);
								if (
									job.existingInvoices &&
									job.existingInvoices.length > 0
								) {
									for (
										let i = 0;
										i < job.existingInvoices.length;
										i++
									) {
										if (
											job.existingInvoices[i] === invoice
										) {
											exist = true;
											break;
										}
									}
								}
								return (
									<Grid container>
										<Grid item xs={4}>
											<Box m={2}>
												<Typography variant="button">
													Invoice{" "}
													{i + 1 + ".) " + invoice}
												</Typography>
											</Box>
										</Grid>
										{exist && (
											<Grid item xs={8}>
												<Box m={2}>
													<Typography variant="button">
														{exist &&
															"This invoice has already been processed before."}
													</Typography>
													<Button
														variant="contained"
														style={{
															backgroundColor:
																"#F8E831",
															marginLeft: "20px",
														}}
													>
														PREVIEW RESULT
													</Button>
													<Button
														variant="contained"
														style={{
															backgroundColor:
																"#F8E831",
															marginLeft: "20px",
														}}
													>
														REEVALUATE
													</Button>
												</Box>
											</Grid>
										)}

										{Object.keys(job.output[invoice]).map(
											(page, j) => (
												<>
													<Grid item xs={12}>
														<Box ml={4} mb={2}>
															<Typography variant="button">
																{"Page : " +
																	page}
															</Typography>
														</Box>
													</Grid>
													{/* <Grid item xs={3}>
													<Box ml={8} mb={2}>
														<Button
															variant="contained"
															style={{
																backgroundColor:
																	"#F8E831",
															}}
															onClick={() => {}}
														>
															Select Original
														</Button>
													</Box>
												</Grid> */}
													<Grid item xs={2}>
														<Box ml={8} mb={2}>
															<Typography variant="button">
																Apply Effect :
															</Typography>
														</Box>
													</Grid>
													{Object.keys(
														job.output[invoice][
															page
														]["images"]
													).map((type) => {
														const url =
															BASE_URL +
															"get_file?path=" +
															job.output[invoice][
																page
															]["images"][type];
														const xlsx =
															BASE_URL +
															"get_file?path=" +
															job.output[invoice][
																page
															]["xlsx"];
														return (
															<Box mr={2}>
																<Button
																	variant="contained"
																	style={{
																		backgroundColor:
																			"#F8E831",
																	}}
																	onClick={() => {
																		setXlsx(
																			xlsx
																		);
																		setImage(
																			url
																		);
																		setOpen(
																			true
																		);
																	}}
																>
																	{type}
																</Button>
															</Box>
														);
													})}
												</>
											)
										)}
										<Grid item xs={12}>
											<Box m={2}>
												<Divider />
											</Box>
										</Grid>
									</Grid>
								);
							})}
						</CardContent>
					</Card>
				)}
			</Paper>
		</>
	);
}

const useStyles = makeStyles({
	paper: {
		width: "100%",
		minWidth: 260,
		display: "inline-block",
	},
	gridList: {
		height: 500,
	},
	titleBar: {
		color: "#FFA500",
		background:
			"linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, " +
			"rgba(0,0,0,0.3) 70%, rgba(0,0,0,0) 100%)",
	},
});
