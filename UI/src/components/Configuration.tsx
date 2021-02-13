import * as React from "react";
import { makeStyles, createStyles } from "@material-ui/styles";
import {
	Paper,
	Grid,
	Theme,
	Typography,
	Button,
	Chip,
	Divider,
	Dialog,
	DialogTitle,
	DialogActions,
	TextField,
	Card,
} from "@material-ui/core";
import axios from "axios";
import { BASE_URL } from "../constants";
import { Configuration as Config } from "../model";

export function Configuration() {
	const classes = useStyles();

	const [loading, setLoading] = React.useState(true);

	const [config, setConfig] = React.useState<Config | undefined>();

	const [openHeading, setOpenHeading] = React.useState(false);

	const [openDetail, setOpenDetail] = React.useState(false);

	const updateNumberOfThreads = async (num_of_threads: number) => {
		setLoading(true);
		let res = await axios.get(
			BASE_URL + `update_num_of_threads/${num_of_threads}`
		);
		setConfig(res.data as Config);
		setLoading(false);
	};

	const getConfiguration = async () => {
		let res = await axios.get(BASE_URL + "/basic_config");
		console.log(res.data);
		setConfig(res.data as Config);
		setLoading(false);
	};

	const deleteHeader = async (header: string) => {
		setLoading(true);
		var bodyFormData = new FormData();
		bodyFormData.append("header", header);
		let res = await axios.post(BASE_URL + "delete_header", bodyFormData);
		setConfig(res.data as Config);
		setLoading(false);
	};

	const addHeader = async (header: string) => {
		setLoading(true);
		var bodyFormData = new FormData();
		bodyFormData.append("header", header);
		let res = await axios.post(BASE_URL + "add_header", bodyFormData);
		setConfig(res.data as Config);
		setLoading(false);
	};

	const deleteDetail = async (detail: string) => {
		setLoading(true);
		var bodyFormData = new FormData();
		bodyFormData.append("detail", detail);
		let res = await axios.post(BASE_URL + "delete_detail", bodyFormData);
		setConfig(res.data as Config);
		setLoading(false);
	};

	const addDetail = async (detail: string) => {
		setLoading(true);
		var bodyFormData = new FormData();
		bodyFormData.append("detail", detail);
		let res = await axios.post(BASE_URL + "add_detail", bodyFormData);
		setConfig(res.data as Config);
		setLoading(false);
	};

	React.useEffect(() => {
		getConfiguration();
	}, []);

	return (
		<Card className={classes.paper}>
			<AddDialog
				open={openHeading}
				title="Add a new Header"
				onClose={() => setOpenHeading(false)}
				onAdd={(heading: string) => {
					setOpenHeading(false);
					addHeader(heading);
				}}
			/>
			<AddDialog
				open={openDetail}
				title="Add a new Detail"
				onClose={() => setOpenDetail(false)}
				onAdd={(detail) => {
					setOpenDetail(false);
					addDetail(detail);
				}}
			/>
			{loading && <div>Loading...</div>}
			{config && (
				<>
					<Grid
						className={classes.configContainer}
						container
						spacing={3}
						style={{ marginBottom: "15px" }}
					>
						<ConfigTitle title="Number Of Threads" />
						<Grid item xs={8}>
							<Grid container spacing={5}>
								<Grid item>
									<Button
										variant="contained"
                                        style={{ backgroundColor: "#F8E831" }}
										onClick={() => {
											updateNumberOfThreads(
												config.num_of_threads - 1
											);
										}}
										disabled={config.num_of_threads <= 1}
									>
										<Typography variant="h5">-</Typography>
									</Button>
								</Grid>
								<Grid item>
									<Typography variant="h5">
										{config.num_of_threads}
									</Typography>
								</Grid>
								<Grid item>
									<Button
										variant="contained"
                                        style={{ backgroundColor: "#F8E831" }}
										onClick={() => {
											updateNumberOfThreads(
												config.num_of_threads + 1
											);
										}}
										disabled={config.num_of_threads > 7}
									>
										<Typography variant="h5">+</Typography>
									</Button>
								</Grid>
							</Grid>
						</Grid>
					</Grid>
					<Divider />
					<Grid
						className={classes.configContainer}
						container
						spacing={3}
						style={{ marginTop: "15px", marginBottom: "15px" }}
					>
						<ConfigTitle title="Header List" />
						<Grid item xs={7}>
							<Grid container>
								{config.headers.map((c, i) => (
									<Chip
										className={classes.chip}
										key={c + i}
										label={c}
										onDelete={() => {
											deleteHeader(c);
										}}
									/>
								))}
							</Grid>
						</Grid>
						<Grid item xs={2}>
							<Button
								variant="contained"
								style={{ backgroundColor: "#F8E831" }}
								onClick={() => {
									setOpenHeading(true);
								}}
							>
								Add Header
							</Button>
						</Grid>
					</Grid>
					<Divider />
					<Grid
						className={classes.configContainer}
						container
						spacing={3}
						style={{ marginTop: "15px" }}
					>
						<ConfigTitle title="Detail List" />
						<Grid item xs={7}>
							<Grid container>
								{config.details.map((d, i) => (
									<Chip
										className={classes.chip}
										key={d + i}
										label={d}
										onDelete={() => {
											deleteDetail(d);
										}}
									/>
								))}
							</Grid>
						</Grid>
						<Grid item xs={2}>
							<Button
								style={{ backgroundColor: "#F8E831" }}
								variant="contained"
								onClick={() => {
									setOpenDetail(true);
								}}
							>
								Add Detail
							</Button>
						</Grid>
					</Grid>
				</>
			)}
		</Card>
	);
}

const AddDialog = (props: {
	open: boolean;
	onClose: () => void;
	onAdd: (value: string) => void;
	title: string;
}) => {
	const { open, onClose, onAdd, title } = props;
	const classes = useStyles();

	const [value, setValue] = React.useState("");

	const handleChange = (event: any) => {
		setValue(event.target.value);
	};

	return (
		<Dialog open={open}>
			<DialogTitle>{title}</DialogTitle>
			<TextField
				id="multiline-flexible"
				multiline
				value={value}
				onChange={handleChange}
				className={classes.textField}
			/>
			<DialogActions>
				<Button color="primary" onClick={() => onAdd(value)}>
					Add
				</Button>
				<Button color="primary" onClick={onClose}>
					Cancel
				</Button>
			</DialogActions>
		</Dialog>
	);
};

const ConfigTitle = (props: { title: string }) => {
	return (
		<Grid item xs={3}>
			<Typography variant="h6">{props.title}</Typography>
		</Grid>
	);
};

const useStyles = makeStyles((theme: Theme) =>
	createStyles({
		root: {
			flexGrow: 1,
		},
		paper: {
			padding: theme.spacing(2),
			color: theme.palette.text.secondary,
		},
		chip: {
			margin: 2,
		},
		configContainer: {
			margin: 50,
		},
		textField: {
			width: "80%",
			margin: 20,
		},
		button: {
			backgroundColor: "#F8E831",
			color: "#ffffff",
		},
	})
);
