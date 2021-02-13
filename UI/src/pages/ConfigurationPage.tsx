import { makeStyles } from "@material-ui/styles";
import * as React from "react";
import { Configuration } from "../components/Configuration";
import { Typography } from "@material-ui/core";

export function ConfigurationPage(){
    const classes = useStyles()

    return (
		<div className={classes.root}>
			<Typography variant="h4" gutterBottom>
				Configuration
			</Typography>
			<div className={classes.centerContainer}>
				<Configuration/>
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