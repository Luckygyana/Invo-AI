import { makeStyles } from "@material-ui/styles";
import * as React from "react";
import { JobView } from "../components";
import { Typography } from "@material-ui/core";

export function JobPage(props: any) {
	const classes = useStyles();

	return (
		<div className={classes.root}>
			<Typography variant="h6" gutterBottom>
				JOB
			</Typography>
			<div className={classes.centerContainer}>
				<JobView jobId={props.match.params.jobId} />
			</div>
		</div>
	);
}

const useStyles = makeStyles({
	root: {
		paddingTop: 20,
		paddingLeft: 15,
		paddingRight: 15,
	},

	centerContainer: {
		flex: 1,
		display: "flex",
		flexDirection: "column",
	},
});

