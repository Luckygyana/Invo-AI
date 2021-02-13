// prettier-ignore
import { AppBar, Badge, Divider, Drawer as DrawerMui, Hidden, IconButton, List, ListItem, ListItemIcon, ListItemText, Toolbar, Typography, useMediaQuery } from "@material-ui/core";
import { Theme } from "@material-ui/core/styles";
import FormatListNumberedIcon from "@material-ui/icons/FormatListNumbered";
import { Settings } from "@material-ui/icons";
import HomeIcon from "@material-ui/icons/Home";
import MenuIcon from "@material-ui/icons/Menu";
import { makeStyles } from "@material-ui/styles";
import * as React from "react";
import { useSelector } from "react-redux";
import { Route, Router } from "react-router-dom";
import { history } from "./configureStore";
import { Todo, Job } from "./model";
import { HomePage, ConfigurationPage, JobListPage, JobPage } from "./pages";
import { RootState } from "./reducers/index";
import { withRoot } from "./withRoot";
import clsx from "clsx";

import { createMuiTheme } from "@material-ui/core/styles";

function Routes(props: { open: boolean }) {
	const classes = useStyles();

	console.log(props.open);

	return (
		<div
			className={clsx(classes.content, {
				[classes.contentShift]: props.open,
			})}
		>
			<Route exact={true} path="/" component={HomePage} />
			<Route exact={true} path="/invoice" component={HomePage} />
			<Route exact={true} path="/jobs" component={JobListPage} />
			<Route
				exact={true}
				path="/configuration"
				component={ConfigurationPage}
			/>
			<Route exact={true} path="/job/:jobId" component={JobPage} />
		</div>
	);
}

function Drawer(props: { jobList: Job[] }) {
	const classes = useStyles();

	return (
		<div>
			<div className={classes.drawerHeader} />
			<Divider />
			<List>
				<ListItem button onClick={() => history.push("/")}>
					<ListItemIcon>
						<HomeIcon />
					</ListItemIcon>
					<ListItemText primary="Invoice" />
				</ListItem>
			</List>
			<Divider />
			<List>
				<ListItem button onClick={() => history.push("/jobs")}>
					<ListItemIcon>
						<JobIcon jobs={props.jobList} />
					</ListItemIcon>
					<ListItemText primary="Job List" />
				</ListItem>
			</List>
			<Divider />
			<List>
				<ListItem button onClick={() => history.push("/configuration")}>
					<ListItemIcon>
						<Settings />
					</ListItemIcon>
					<ListItemText primary="Configuration" />
				</ListItem>
			</List>
			<Divider />
		</div>
	);
}

function App() {
	const classes = useStyles();
	const todoList = useSelector((state: RootState) => state.todoList);
	const [open, setOpen] = React.useState(true);

	const toggleDrawer = () => {
		setOpen(!open);
	};

	const contentStyle = {
		transition: "margin-left 450ms cubic-bezier(0.23, 1, 0.32, 1)",
	};

	// if (this.state.drawerOpen) {
	//   contentStyle.marginLeft = 256;
	// }

	return (
		<Router history={history}>
			<div className={classes.root}>
				<div className={classes.appFrame}>
					<AppBar className={classes.appBar} color="secondary">
						<Toolbar>
							<IconButton
								color="inherit"
								aria-label="open drawer"
								onClick={toggleDrawer}
								className={classes.navIconHide}
							>
								<MenuIcon />
							</IconButton>
							<Typography variant="h6" color="inherit">
								Invo - AI : Samsung Innovation Challenge 2020
							</Typography>
						</Toolbar>
					</AppBar>
					<DrawerMui
						variant="persistent"
						anchor={"left"}
						open={open}
						classes={{
							paper: classes.drawerPaper,
						}}
						ModalProps={{
							keepMounted: true, // Better open performance on mobile.
						}}
					>
						<Drawer jobList={[]} />
					</DrawerMui>
					<Routes open={open} />
				</div>
			</div>
		</Router>
	);
}

function JobIcon(props: { jobs: Job[] }) {
	let uncompletedJobs = props.jobs.filter((j) => j.status === "processing");

	if (uncompletedJobs.length > 0) {
		return (
			<Badge color="secondary" badgeContent={uncompletedJobs.length}>
				<FormatListNumberedIcon />
			</Badge>
		);
	} else {
		return <FormatListNumberedIcon />;
	}
}

function TodoIcon(props: { todoList: Todo[] }) {
	let uncompletedTodos = props.todoList.filter((t) => t.completed === false);

	if (uncompletedTodos.length > 0) {
		return (
			<Badge color="secondary" badgeContent={uncompletedTodos.length}>
				<FormatListNumberedIcon />
			</Badge>
		);
	} else {
		return <FormatListNumberedIcon />;
	}
}

const drawerWidth = 240;
const useStyles = makeStyles((theme: Theme) => ({
	root: {
		width: "100%",
		zIndex: 1,
		overflow: "hidden",
	},
	appFrame: {
		position: "relative",
		display: "flex",
		width: "100%",
	},
	appBar: {
		zIndex: theme.zIndex.drawer + 1,
		position: "absolute",
		backgroundColor: "#000000",
	},
	navIconHide: {
		[theme.breakpoints.up("md")]: {
			display: "none",
		},
	},
	drawerHeader: { ...theme.mixins.toolbar },
	drawerPaper: {
		width: 250,
		backgroundColor: theme.palette.background.default,
		[theme.breakpoints.up("md")]: {
			width: drawerWidth,
			position: "relative",
			height: "100%",
		},
	},
	content: {
		backgroundColor: theme.palette.background.default,
		width: "100%",
		marginTop: 56,
		[theme.breakpoints.up("sm")]: {
			height: "calc(100% - 64px)",
			marginTop: 64,
		},
	},
	contentShift: {
		transition: theme.transitions.create("margin", {
			easing: theme.transitions.easing.easeOut,
			duration: theme.transitions.duration.enteringScreen,
		}),
		marginLeft: 240,
	},
}));

export default withRoot(App);
