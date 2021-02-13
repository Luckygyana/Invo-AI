import {
	Checkbox,
	IconButton,
	Paper,
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableRow,
	Tab,
} from "@material-ui/core";
import * as React from "react";
import { makeStyles } from "@material-ui/styles";
import { Job } from "../model";
import Axios from "axios";
import { BASE_URL } from "../constants";
import { history } from "../configureStore";

export function JobTable() {
	const classes = useStyles();

	const [loading, setLoading] = React.useState(true);
	const [jobs, setJobs] = React.useState<Job[]>([]);

	React.useEffect(() => {
		getJobs();
	}, []);

	const getJobs = async () => {
		setLoading(true);
		const res = await Axios.get(BASE_URL + "jobs");
		const data = res.data;
		console.log(data);
		setJobs(data);
		setLoading(false);
	};

	return (
		<Paper className={classes.paper}>
			{loading && "Loading..."}
			<Table className={classes.table}>
				<TableHead>
					<TableRow>
						<TableCell padding="default">Job ID</TableCell>
						<TableCell padding="default">Status</TableCell>
						<TableCell padding="default">Files</TableCell>
					</TableRow>
				</TableHead>
				<TableBody>
					{jobs.map((j: Job) => {
						return (
							<TableRow
								key={j._id.$oid}
								hover
								onClick={(event) => {
									history.push("/job/" + j._id.$oid);
								}}
							>
								<TableCell padding="default">
									{j._id.$oid}
								</TableCell>
								<TableCell padding="default">
									{j.status}
								</TableCell>
								<TableCell padding="default">
									{j.files.length}
								</TableCell>
							</TableRow>
						);
					})}
				</TableBody>
			</Table>
		</Paper>
	);
}

const useStyles = makeStyles({
	paper: {
		width: "100%",
		minWidth: 260,
		display: "inline-block",
	},
	table: {
		width: "100%",
	},
});
