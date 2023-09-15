# Generated by Django 4.0.3 on 2023-04-26 11:42

import datetime
from django.db import migrations, models
import django.utils.timezone
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('service_rest', '0004_delete_customervo_appointment_vip_status'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='appointment',
            options={'ordering': ('appt_date', 'appt_time')},
        ),
        migrations.RemoveField(
            model_name='appointment',
            name='date_time',
        ),
        migrations.AddField(
            model_name='appointment',
            name='appt_date',
            field=models.DateField(default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='appointment',
            name='appt_time',
            field=models.TimeField(default=datetime.datetime(2023, 4, 26, 11, 42, 2, 148199, tzinfo=utc)),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='appointment',
            name='updated',
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AlterField(
            model_name='technician',
            name='employee_id',
            field=models.CharField(max_length=25, unique=True),
        ),
        migrations.AlterField(
            model_name='technician',
            name='first_name',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='technician',
            name='last_name',
            field=models.CharField(max_length=100),
        ),
    ]
