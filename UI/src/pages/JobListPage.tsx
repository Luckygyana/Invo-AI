import { Button, Grid, Typography } from "@material-ui/core";
import { Theme } from "@material-ui/core/styles";
import { makeStyles } from "@material-ui/styles";
import * as React from "react";
import { JobTable } from "../components";

export function JobListPage() {
	const classes = useStyles();

	return (
		<Grid container className={classes.root}>
			<Grid item xs={6}>
				<Typography variant="h6" gutterBottom>
					JOB LIST
				</Typography>
			</Grid>
			<Grid item xs={12}>
				<JobTable />
			</Grid>
		</Grid>
	);
}

const useStyles = makeStyles((theme: Theme) => ({
	root: {
		padding: 20,
		[theme.breakpoints.down("md")]: {
			paddingTop: 50,
			paddingLeft: 15,
			paddingRight: 15,
		},
	},

	buttonContainer: {
		width: "100%",
		display: "flex",
		justifyContent: "flex-end",
	},

	button: {
		marginBottom: 15,
	},
}));
