# VPN Connection Guide for Windows

This runbook explains how to connect to the corporate VPN from a Windows laptop.

## Prerequisites

- An active internet connection.
- Installed corporate VPN client (for example, Cisco AnyConnect).
- Valid corporate VPN account.

## Steps

1. Open the **Cisco AnyConnect Secure Mobility Client**.
2. In the **VPN** field, enter the server address: **vpn.company.com**.
3. Click **Connect**.
4. In the authentication window:
   - Enter your corporate username.
   - Enter your corporate password.
5. If multi-factor authentication (MFA) is enabled:
   - Approve the push notification in the mobile app,
   - or enter the one-time code from SMS or authenticator app.
6. Wait until the VPN status changes to **Connected**.
7. Once connected, verify that you can open internal resources (for example, https://intranet.company.com).

## Troubleshooting

- If the client shows **Authentication failed**:
  - Verify that you are using the correct username and password.
  - Try logging into webmail with the same credentials.
  - If login fails everywhere, your account might be locked; contact IT Support.
- If VPN connects but internal sites are still unavailable:
  - Disconnect and reconnect the VPN.
  - Check that your firewall or antivirus does not block VPN traffic.
  - Try another network (for example, mobile hotspot) to exclude local network issues.
